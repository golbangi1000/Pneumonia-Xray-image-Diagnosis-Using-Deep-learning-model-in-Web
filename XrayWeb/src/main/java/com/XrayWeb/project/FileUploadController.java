package com.XrayWeb.project;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;
import com.XrayWeb.project.DeepSystemCall;

@Controller
@RequestMapping("/common")
public class FileUploadController{
	
	@Autowired 
	xray_dataService xray_dataService;
	
	@Autowired
	xray_dataService XService;
	
	@RequestMapping("/upload")
	public ModelAndView upload(@RequestParam("uploadFile") MultipartFile file, HttpServletRequest request, ModelAndView mv, Model model) throws IllegalStateException, IOException {
		
		String path = "C:/Users/kenja/Desktop/XrayWeb/XrayWeb/src/main/webapp/resources/upload/file/";
		String path1 = path + request.getParameter("member_id");
		
		File Folder = new File(path1);
		
		if (!Folder.exists()) {
			try{
			    Folder.mkdir(); 
			    System.out.println("Make folder.");
		        } 
		        catch(Exception e){
			    e.getStackTrace();
			}        
	         }else {
			System.out.println("Folder Exists.");
		}
		
		if(!file.getOriginalFilename().isEmpty()) {
			file.transferTo(new File(path1, file.getOriginalFilename()));
			
			model.addAttribute("msg", "File uploaded successfully.");
		}else {
			model.addAttribute("msg", "Please select a valid mediaFile..");
		}
		
		Map<String, Object> file_Map= new HashMap<String, Object>();
		
		file_Map.put("member_Id", request.getParameter("member_id")); 
		//file_Map.put("Image_path", request.getParameter("member_Id")+"/"+file.getOriginalFilename()); 
		file_Map.put("Image_path", file.getOriginalFilename()); 
		xray_dataService.xray_datacreate(file_Map);
		
		
		//detail list gain
		ModelAndView mav = new ModelAndView(); 
		
		mav.addObject("member_Id", request.getParameter("member_id"));
		
		
		HashMap<String,Object> map1 = new HashMap<String,Object>();

		map1.put("member_Id",request.getParameter("member_id"));
		
		System.out.println(map1);
		List<Map<String, Object>> lists = XService.xray_dataList(map1);
		
		System.out.println(lists);
		mav.addObject("lists", lists);
		mav.setViewName("member/detail");
		
		DeepSystemCall DL_SCall = new DeepSystemCall();
		
		System.out.println("test1");
		DL_SCall.linuxstart(request.getParameter("member_id"),file.getOriginalFilename());
		System.out.println("test2");
		return mav;
	}
}